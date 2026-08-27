## Repository with reinforcement learning 

The repository is experimental and undergoing changes. The basic idea is to experiment some distinct topologies from the literature. 

Currently we have an Agent, that is composed of:
1) Memory() - the memory is an external storage system 


2) Perception(frame_stack, img_size, d_model)
3) Projection(self.perception) 
4) DreamerCenter(d_model, world.n_actions) # Expects (z, action)
5) Attention nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
6) Cortex(d_model, world.n_actions)
7) RewardCenter(d_model)
        